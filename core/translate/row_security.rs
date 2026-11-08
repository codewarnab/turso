//! Row-level security limits which rows of a table a role may access.
//!
//! The statements that read and write rows call the hooks of this module,
//! which ask the dialect whether row-level security applies to the role,
//! [`crate::Dialect::row_security_applies`], and limit the statement to the
//! rows the role may access. The role is the one the statement is compiled
//! for, [`Resolver::row_security_role`].
//!
//! Which tables have row-level security, and their policies, are the
//! dialect's schema state, like the owner of a table. The module compiles the
//! statements that change them into a [`RelationChange`] for the dialect to
//! record.

use std::sync::Arc;

use turso_parser::ast;

use crate::{
    bail_parse_error,
    dialect::RelationChange,
    function::Func,
    namespace::{NamespaceId, NamespaceRef},
    schema::{BTreeTable, Schema, Table},
    security::permission::{Action, Object, PermissionDenied},
    security::policy::{Command, Policy, RowPolicy},
    security::roles::{RoleCatalog, RoleId},
    translate::{
        emitter::Resolver,
        expr::{bind_and_rewrite_expr, walk_expr, walk_expr_mut, BindingBehavior, WalkControl},
        plan::{JoinOrigin, JoinedTable, TableReferences, WhereTerm, WhereTermOrigin},
        planner::break_predicate_at_and_boundaries,
        roles,
        schema::emit_schema_cookie,
    },
    util::normalize_ident,
    vdbe::builder::ProgramBuilder,
    vtab::{VirtualTable, VirtualTableType},
    Connection, Dialect, LimboError, Result, MAIN_DB_ID,
};

/// Limits the rows of the tables in `table_references` that the statement
/// reads, updates or deletes to the rows the role may access.
///
/// Each such table gets the condition its rows must pass as
/// [`JoinedTable::row_filter`], which the loop over the table checks before
/// any other condition, so no expression of the statement sees a row the
/// role may not access. On the right side of a LEFT JOIN a row that fails it
/// reads as NULLs, like a row that does not exist.
///
/// The parts of the condition are also added to `where_clause` so that the
/// optimizer can use them, for example to seek an index on the column a
/// policy compares with the role. Rows that fail them were already skipped.
pub(crate) fn filter_rows(
    command: Command,
    table_references: &mut TableReferences,
    where_clause: &mut Vec<WhereTerm>,
    resolver: &Resolver,
) -> Result<()> {
    let mut terms = Vec::new();
    for index in 0..table_references.joined_tables().len() {
        let table = &table_references.joined_tables()[index];
        let Table::BTree(btree) = &table.table else {
            continue;
        };
        let filter = match policy(resolver, table.namespace, &btree.declared_name, command) {
            Policy::Allow => continue,
            Policy::Deny => no_rows(),
            Policy::Filter(policies) => policy_filter(&policies, table, btree, resolver)?,
        };
        reject_joins_that_keep_right_rows(table_references, &btree.declared_name)?;
        let origin = if table
            .join_info
            .as_ref()
            .is_some_and(|join_info| join_info.is_outer())
        {
            WhereTermOrigin::Join(JoinOrigin::Outer(table.internal_id))
        } else {
            WhereTermOrigin::Where
        };
        let mut parts: Vec<ast::Expr> = Vec::new();
        break_predicate_at_and_boundaries(&filter, &mut parts);
        terms.extend(parts.into_iter().map(|expr| WhereTerm {
            expr,
            origin,
            consumed: false,
        }));
        let used_columns = columns_of_table(&filter, table.internal_id)?;
        let table = &mut table_references.joined_tables_mut()[index];
        for column in used_columns {
            table.mark_column_used(column);
        }
        table.row_filter = Some(filter);
    }
    where_clause.splice(0..0, terms);
    Ok(())
}

/// Checks that the role may insert rows into `table`.
pub(crate) fn check_insert(
    table: &BTreeTable,
    namespace: NamespaceRef,
    resolver: &Resolver,
) -> Result<()> {
    match policy(resolver, namespace, &table.declared_name, Command::Insert) {
        Policy::Allow => Ok(()),
        Policy::Deny | Policy::Filter(_) => Err(LimboError::PermissionDenied(Box::new(
            PermissionDenied::policy(Object::Table(table.declared_name.clone())),
        ))),
    }
}

/// Checks that the role may read the virtual table. A virtual table that
/// shows the database file would show the rows that row-level security
/// hides, so only a superuser may read it, like a PRAGMA.
pub(crate) fn check_virtual_table(table: &VirtualTable, resolver: &Resolver) -> Result<()> {
    let reads_raw_storage = matches!(
        &table.vtab_type,
        VirtualTableType::Internal(internal) if internal.read().reads_raw_storage()
    );
    if reads_raw_storage && resolver.role_is_not_superuser {
        return Err(LimboError::PermissionDenied(Box::new(
            PermissionDenied::privilege(Action::Read, Object::Table(table.name.clone())),
        )));
    }
    Ok(())
}

/// Checks that the current role may open a BLOB handle on a row of `table`
/// of the root namespace, which reads and writes the row without a
/// statement, so without the conditions of policies.
pub(crate) fn check_blob_open(
    connection: &Arc<Connection>,
    table_name: &str,
    read_write: bool,
) -> Result<()> {
    let (command, action) = if read_write {
        (Command::Update, Action::Write)
    } else {
        (Command::Select, Action::Read)
    };
    let dialect = connection.dialect();
    let policy = connection.with_schema(MAIN_DB_ID, |schema| {
        policy_for(
            dialect.as_ref(),
            &schema.roles,
            schema,
            connection.current_role(),
            NamespaceId::ROOT,
            table_name,
            command,
        )
    });
    match policy {
        Policy::Allow => Ok(()),
        Policy::Deny | Policy::Filter(_) => Err(LimboError::PermissionDenied(Box::new(
            PermissionDenied::privilege(action, Object::Table(table_name.to_string())),
        ))),
    }
}

/// Which rows of table `table_name` of `namespace` the role may access for
/// `command`. Inside a view, the view's owner reads the table, as for
/// privileges. Whether the table has row-level security is a question for
/// the table's own database, so a table of an attached database with
/// row-level security is limited too. The roles are those of the main
/// database, which the role was set from.
fn policy(
    resolver: &Resolver,
    namespace: NamespaceRef,
    table_name: &str,
    command: Command,
) -> Policy {
    let roles = resolver.schema().roles.clone();
    // A statement the engine compiles as the superuser stays unlimited
    // inside a view.
    let role = if roles.is_superuser(resolver.row_security_role) {
        resolver.row_security_role
    } else {
        roles::table_reader(resolver, resolver.row_security_role)
    };
    resolver.with_schema(namespace.database_id, |schema| {
        policy_for(
            resolver.catalog.dialect.as_ref(),
            &roles,
            schema,
            role,
            namespace.id,
            table_name,
            command,
        )
    })
}

/// Which rows of table `table_name` of `namespace` in `schema` `role` may
/// access for `command`. Row-level security does not apply to superusers.
///
/// A role reads the rows that pass at least one permissive policy and every
/// restrictive policy for SELECT that apply to it. Writes by roles are not
/// supported yet, so a table with row-level security refuses them, whatever
/// its policies say.
fn policy_for(
    dialect: &dyn Dialect,
    roles: &RoleCatalog,
    schema: &Schema,
    role: RoleId,
    namespace: NamespaceId,
    table_name: &str,
    command: Command,
) -> Policy {
    let table_name = normalize_ident(table_name);
    if roles.is_superuser(role)
        || !dialect.row_security_applies(schema, namespace, &table_name, role)
    {
        return Policy::Allow;
    }
    if command != Command::Select {
        return Policy::Deny;
    }
    // A policy without USING says nothing about which rows a role may read.
    let policies: Vec<RowPolicy> = dialect
        .row_security_policies(schema, namespace, &table_name)
        .iter()
        .filter(|policy| policy.applies_to(command, role) && policy.using_expr.is_some())
        .cloned()
        .collect();
    if !policies.iter().any(|policy| policy.permissive) {
        return Policy::Deny;
    }
    Policy::Filter(policies)
}

/// The condition that the rows of `table` must pass under `policies`: the
/// USING conditions of the permissive policies combined with OR, and of the
/// restrictive ones with AND.
fn policy_filter(
    policies: &[RowPolicy],
    table: &JoinedTable,
    btree: &BTreeTable,
    resolver: &Resolver,
) -> Result<ast::Expr> {
    let mut permissive = None;
    let mut restrictive = None;
    for policy in policies {
        let using_expr = policy
            .using_expr
            .as_ref()
            .expect("policy_for keeps only policies with USING");
        let condition = bind_policy_expr(using_expr, table, btree, resolver)?;
        if policy.permissive {
            permissive = Some(combine(permissive, ast::Operator::Or, condition));
        } else {
            restrictive = Some(combine(restrictive, ast::Operator::And, condition));
        }
    }
    let permissive = permissive.expect("policy_for keeps a permissive policy");
    Ok(match restrictive {
        None => permissive,
        Some(restrictive) => combine(Some(permissive), ast::Operator::And, restrictive),
    })
}

fn combine(left: Option<ast::Expr>, operator: ast::Operator, right: ast::Expr) -> ast::Expr {
    match left {
        None => right,
        Some(left) => ast::Expr::Binary(
            Box::new(parenthesized(left)),
            operator,
            Box::new(parenthesized(right)),
        ),
    }
}

fn parenthesized(expr: ast::Expr) -> ast::Expr {
    ast::Expr::Parenthesized(vec![Box::new(expr)])
}

/// Binds a policy condition to `table`, the table it protects. The condition
/// sees only that table, whatever the statement around it has in scope.
fn bind_policy_expr(
    expr: &ast::Expr,
    table: &JoinedTable,
    btree: &BTreeTable,
    resolver: &Resolver,
) -> Result<ast::Expr> {
    let mut policy_table = table.clone();
    policy_table.identifier = normalize_ident(&btree.declared_name);
    policy_table.has_alias = false;
    policy_table.join_info = None;
    let mut table_references = TableReferences::new(vec![policy_table], Vec::new());
    let mut expr = expr.clone();
    bind_and_rewrite_expr(
        &mut expr,
        Some(&mut table_references),
        None,
        resolver,
        BindingBehavior::ResultColumnsNotAllowed,
    )?;
    Ok(expr)
}

/// The columns of the table `internal_id` that the bound `expr` reads.
fn columns_of_table(expr: &ast::Expr, internal_id: ast::TableInternalId) -> Result<Vec<usize>> {
    let mut columns = Vec::new();
    walk_expr(expr, &mut |expr| {
        if let ast::Expr::Column { table, column, .. } = expr {
            if *table == internal_id {
                columns.push(*column);
            }
        }
        Ok(WalkControl::Continue)
    })?;
    Ok(columns)
}

/// A RIGHT or FULL JOIN emits the unmatched rows of its right side after the
/// join loop, without the filters of the tables on its left. A filter on
/// either side would then hide too many rows or too few.
fn reject_joins_that_keep_right_rows(
    table_references: &TableReferences,
    table_name: &str,
) -> Result<()> {
    for join_info in table_references
        .joined_tables()
        .iter()
        .filter_map(|table| table.join_info.as_ref())
    {
        if join_info.is_full_outer() {
            bail_parse_error!(
                "FULL JOIN with table \"{table_name}\" that has row-level security is not supported"
            );
        }
        if join_info.keeps_right_rows() {
            bail_parse_error!(
                "RIGHT JOIN with table \"{table_name}\" that has row-level security is not supported"
            );
        }
    }
    Ok(())
}

/// A condition that no row passes.
fn no_rows() -> ast::Expr {
    ast::Expr::Literal(ast::Literal::Numeric("0".to_string()))
}

/// ALTER TABLE ... ENABLE ROW LEVEL SECURITY, or DISABLE when `enabled` is
/// false. The dialect records it with the table, so it follows the table
/// when the table is renamed and goes away when the table is dropped.
/// [`crate::translate::roles::check_statement_privileges`] lets the table's
/// owner and superusers run it.
pub(crate) fn translate_set_row_security(
    qualified_name: &ast::QualifiedName,
    enabled: bool,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<Connection>,
) -> Result<()> {
    let (namespace, table) = row_security_table(qualified_name, resolver)?;
    translate_change(
        RelationChange::SetRowSecurity {
            namespace: namespace.id,
            name: normalize_ident(&table.declared_name),
            enabled,
        },
        resolver,
        program,
        connection,
    )
}

/// CREATE POLICY. The dialect stores the policy with the table, so it
/// follows the table when the table is renamed and goes away when the table
/// is dropped.
///
/// The conditions may refer to the columns of the table, written as `column`
/// or `table.column`, and call scalar functions. `(SELECT <expr>)` without
/// FROM, which Supabase recommends for function calls, stands for `<expr>`.
/// The conditions are stored with unqualified column names.
/// [`crate::translate::roles::check_statement_privileges`] lets the table's
/// owner and superusers run it.
pub(crate) fn translate_create_policy(
    policy: &ast::CreatePolicy,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<Connection>,
) -> Result<()> {
    let (namespace, table) = row_security_table(&policy.tbl_name, resolver)?;
    if policy.check_expr.is_some()
        && matches!(
            policy.command,
            ast::PolicyCommand::Select | ast::PolicyCommand::Delete
        )
    {
        bail_parse_error!("WITH CHECK cannot be applied to SELECT or DELETE");
    }
    if policy.using_expr.is_some() && policy.command == ast::PolicyCommand::Insert {
        bail_parse_error!("only WITH CHECK expression allowed for INSERT");
    }
    let mut stored = policy.clone();
    stored.policy_name = ast::Name::exact(normalize_ident(policy.policy_name.as_str()));
    for expr in [&mut stored.using_expr, &mut stored.check_expr]
        .into_iter()
        .flatten()
    {
        prepare_policy_expr(expr, &table, resolver)?;
    }
    translate_change(
        RelationChange::CreatePolicy {
            namespace: namespace.id,
            name: normalize_ident(&table.declared_name),
            policy: Box::new(stored),
        },
        resolver,
        program,
        connection,
    )
}

/// DROP POLICY. With IF EXISTS, a missing table or policy is not an error,
/// as in PostgreSQL.
pub(crate) fn translate_drop_policy(
    policy_name: &ast::Name,
    tbl_name: &ast::QualifiedName,
    if_exists: bool,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<Connection>,
) -> Result<()> {
    let table_exists = resolver
        .resolve_existing_table(tbl_name)
        .is_ok_and(|namespace| {
            resolver.with_schema(namespace.database_id, |schema| {
                schema
                    .get_table(namespace.id, tbl_name.name.as_str())
                    .is_some()
            })
        });
    if if_exists && !table_exists {
        return Ok(());
    }
    let (namespace, table) = row_security_table(tbl_name, resolver)?;
    translate_change(
        RelationChange::DropPolicy {
            namespace: namespace.id,
            name: normalize_ident(&table.declared_name),
            policy_name: normalize_ident(policy_name.as_str()),
            if_exists,
        },
        resolver,
        program,
        connection,
    )
}

/// Policies refer to the columns of their table by name, so a column that a
/// policy uses cannot be renamed, dropped or redefined.
pub(crate) fn reject_change_of_policy_column(
    resolver: &Resolver,
    namespace: NamespaceRef,
    table_name: &str,
    column_name: &str,
) -> Result<()> {
    if namespace.database_id != MAIN_DB_ID {
        return Ok(());
    }
    let table_name = normalize_ident(table_name);
    let column_name = normalize_ident(column_name);
    let used_by = resolver
        .catalog
        .dialect
        .row_security_policies(resolver.schema(), namespace.id, &table_name)
        .iter()
        .find(|policy| policy_uses_column(policy, &column_name))
        .map(|policy| policy.name.clone());
    if let Some(policy_name) = used_by {
        bail_parse_error!(
            "cannot change column \"{column_name}\" of table \"{table_name}\" because policy \"{policy_name}\" uses it"
        );
    }
    Ok(())
}

fn policy_uses_column(policy: &RowPolicy, column_name: &str) -> bool {
    let mut uses_column = false;
    for expr in [&policy.using_expr, &policy.check_expr]
        .into_iter()
        .flatten()
    {
        walk_expr(expr, &mut |expr| {
            if let ast::Expr::Id(name) = expr {
                uses_column |= normalize_ident(name.as_str()) == column_name;
            }
            Ok(WalkControl::Continue)
        })
        .expect("the walk never fails");
    }
    uses_column
}

/// The table of the main database that row-level security statements name.
/// Views, materialized views and the engine's own tables cannot have
/// row-level security.
fn row_security_table(
    qualified_name: &ast::QualifiedName,
    resolver: &Resolver,
) -> Result<(NamespaceRef, Arc<BTreeTable>)> {
    let namespace = resolver.resolve_existing_table(qualified_name)?;
    let table_name = normalize_ident(qualified_name.name.as_str());
    if namespace.database_id != MAIN_DB_ID {
        bail_parse_error!("row-level security is only supported for tables in the main database");
    }
    let schema = resolver.schema();
    let is_view = schema.get_view(namespace.id, &table_name).is_some()
        || schema
            .get_materialized_view(namespace.id, &table_name)
            .is_some();
    if is_view {
        bail_parse_error!("\"{table_name}\" is not a table");
    }
    match schema.get_btree_table(namespace.id, &table_name) {
        Some(table) => Ok((namespace, table)),
        None if schema.get_table(namespace.id, &table_name).is_some() => {
            bail_parse_error!("\"{table_name}\" is not a table")
        }
        None => bail_parse_error!(
            "relation \"{}\" does not exist",
            crate::util::table_name_for_error(qualified_name)
        ),
    }
}

/// Records `change` through the dialect, in a write transaction that bumps
/// the schema cookie so that other connections load it.
fn translate_change(
    change: RelationChange,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<Connection>,
) -> Result<()> {
    let schema = resolver.schema();
    program.begin_write_on_database(MAIN_DB_ID, schema.schema_version)?;
    resolver
        .catalog
        .dialect
        .translate_relation_changes(program, schema, connection, &[change])?;
    emit_schema_cookie(resolver, program);
    Ok(())
}

/// Checks a policy condition of `table` and puts it in the form it is stored
/// in: `(SELECT <expr>)` without FROM becomes `(<expr>)`, and columns lose
/// their table name, so the condition still holds after the table is
/// renamed. Column names are quoted, so a column named like a keyword, such
/// as `true`, is still read as the column.
fn prepare_policy_expr(
    expr: &mut ast::Expr,
    table: &BTreeTable,
    resolver: &Resolver,
) -> Result<()> {
    replace_scalar_selects(expr)?;
    let table_name = normalize_ident(&table.declared_name);
    walk_expr_mut(expr, &mut |expr: &mut ast::Expr| {
        match expr {
            ast::Expr::Qualified(qualifier, name) => {
                if normalize_ident(qualifier.as_str()) != table_name {
                    bail_parse_error!(
                        "a policy of table \"{table_name}\" can only refer to the columns of that table"
                    );
                }
                *expr = ast::Expr::Id(name.clone());
            }
            ast::Expr::DoublyQualified(..) => bail_parse_error!(
                "a policy of table \"{table_name}\" can only refer to the columns of that table"
            ),
            ast::Expr::Subquery(_) | ast::Expr::Exists(_) | ast::Expr::InSelect { .. } => {
                bail_parse_error!("subqueries in policies are not supported")
            }
            ast::Expr::Variable(_) => {
                bail_parse_error!("there is no parameter $1")
            }
            _ => {}
        }
        Ok(WalkControl::Continue)
    })?;
    walk_expr(expr, &mut |expr| {
        match expr {
            ast::Expr::Id(name) => {
                let Some((_, column)) = table.get_column(name.as_str()) else {
                    bail_parse_error!("column \"{}\" does not exist", name.as_str());
                };
                if column.is_virtual_generated() {
                    bail_parse_error!("policies cannot use generated column \"{}\"", name.as_str());
                }
            }
            ast::Expr::FunctionCall { name, args, .. } => {
                check_policy_function(name.as_str(), args.len(), resolver)?;
            }
            ast::Expr::FunctionCallStar { name, .. } => {
                check_policy_function(name.as_str(), 0, resolver)?;
            }
            _ => {}
        }
        Ok(WalkControl::Continue)
    })?;
    walk_expr_mut(expr, &mut |expr: &mut ast::Expr| {
        if let ast::Expr::Id(name) = expr {
            *name = ast::Name::from_string(format!("\"{}\"", name.as_str().replace('"', "\"\"")));
        }
        Ok(WalkControl::Continue)
    })?;
    Ok(())
}

/// A policy condition is checked on one row at a time, so it can only call
/// scalar functions, as in PostgreSQL.
fn check_policy_function(name: &str, arg_count: usize, resolver: &Resolver) -> Result<()> {
    match resolver.resolve_function(name, arg_count)? {
        None => bail_parse_error!("function {name} does not exist"),
        Some(Func::Agg(_)) => {
            bail_parse_error!("aggregate functions are not allowed in policy expressions")
        }
        Some(Func::Window(_)) => {
            bail_parse_error!("window functions are not allowed in policy expressions")
        }
        Some(_) => Ok(()),
    }
}

/// Replaces each `(SELECT <expr>)` without FROM or any other clause by
/// `(<expr>)`.
fn replace_scalar_selects(expr: &mut ast::Expr) -> Result<()> {
    walk_expr_mut(expr, &mut |expr: &mut ast::Expr| {
        if let ast::Expr::Subquery(select) = expr {
            if let Some(value) = single_value_of(select) {
                *expr = parenthesized(value.clone());
            }
        }
        Ok(WalkControl::Continue)
    })?;
    Ok(())
}

/// The expression of `SELECT <expr>` without FROM or any other clause.
fn single_value_of(select: &ast::Select) -> Option<&ast::Expr> {
    if select.with.is_some()
        || !select.body.compounds.is_empty()
        || !select.order_by.is_empty()
        || select.limit.is_some()
    {
        return None;
    }
    match &select.body.select {
        ast::OneSelect::Select {
            distinctness: None,
            columns,
            from: None,
            where_clause: None,
            group_by: None,
            window_clause,
        } if window_clause.is_empty() => match columns.as_slice() {
            [ast::ResultColumn::Expr(expr, _)] => Some(expr),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rustc_hash::FxHashMap;
    use turso_parser::ast;

    use super::{policy, Command, Policy};
    use crate::dialect::{Dialect, EngineDialect};
    use crate::function::Func;
    use crate::namespace::{NamespaceId, NamespaceRef};
    use crate::schema::{BTreeTable, Schema};
    use crate::security::roles::{RoleCatalog, RoleId};
    use crate::translate::emitter::{DoubleQuotedDml, Resolver};
    use crate::{Database, DatabaseCatalog, MemoryIO, Qualifiers, RwLock, SymbolTable, Value};

    /// The engine dialect, except that row-level security applies to every
    /// table of a schema with dialect schema state.
    struct RowSecurityInSchemasWithState;

    impl Dialect for RowSecurityInSchemasWithState {
        fn name(&self) -> &'static str {
            EngineDialect.name()
        }

        fn parse(&self, sql: &str) -> crate::Result<(Option<ast::Cmd>, usize)> {
            EngineDialect.parse(sql)
        }

        fn parse_table_sql(&self, sql: &str, root_page: i64) -> crate::Result<BTreeTable> {
            EngineDialect.parse_table_sql(sql, root_page)
        }

        fn parse_table_sql_ast(&self, sql: &str) -> crate::Result<ast::Stmt> {
            EngineDialect.parse_table_sql_ast(sql)
        }

        fn table_sql_for_replay(&self, sql: &str) -> crate::Result<String> {
            EngineDialect.table_sql_for_replay(sql)
        }

        fn format_table_sql(
            &self,
            input: &str,
            tbl_name: &ast::QualifiedName,
            body: &ast::CreateTableBody,
        ) -> crate::Result<String> {
            EngineDialect.format_table_sql(input, tbl_name, body)
        }

        fn register_catalog(
            &self,
            schema: &mut Schema,
            enable_custom_types: bool,
        ) -> crate::Result<()> {
            EngineDialect.register_catalog(schema, enable_custom_types)
        }

        fn resolve_function(&self, name: &str, arg_count: usize) -> crate::Result<Option<Func>> {
            EngineDialect.resolve_function(name, arg_count)
        }

        fn row_security_applies(
            &self,
            schema: &Schema,
            _namespace: NamespaceId,
            _name: &str,
            _role: RoleId,
        ) -> bool {
            schema.dialect_schema.is_some()
        }
    }

    /// Whether a table of an attached database has row-level security is a
    /// question for the attached database, not for the main database.
    #[test]
    fn table_of_an_attached_database_with_row_security_denies_a_role() {
        let roles = RoleCatalog::new(&[vec![
            Value::from_i64(1),
            Value::build_text("alice"),
            Value::from_i64(0),
        ]])
        .unwrap();
        let alice = roles.get_by_name("alice").unwrap().id;
        let mut main = Schema::default();
        main.roles = Arc::new(roles);
        let mut attached = Schema::default();
        attached.dialect_schema = Some(Arc::new(()));
        let attached_database_id = 2;
        let mut database_schemas = FxHashMap::default();
        database_schemas.insert(attached_database_id, Arc::new(attached));
        let database_schemas = RwLock::new(database_schemas);
        let temp_database = RwLock::new(None);
        let attached_databases = RwLock::new(DatabaseCatalog::new());
        let qualifiers = Qualifiers::default();
        let syms = SymbolTable::new();
        let resolver = Resolver::new(
            &main,
            &database_schemas,
            &temp_database,
            &attached_databases,
            &qualifiers,
            &syms,
            true,
            DoubleQuotedDml::Enabled,
            Arc::new(RowSecurityInSchemasWithState),
            None,
            alice,
        );

        let policy = policy(
            &resolver,
            NamespaceRef::root_of(attached_database_id),
            "t",
            Command::Select,
        );

        assert_eq!(policy, Policy::Deny);
    }

    /// A materialized view stores the rows of its query when it is refreshed,
    /// so policies on it could not limit them.
    #[test]
    fn row_security_statements_reject_a_materialized_view() {
        let db = Database::open_file_with_flags(
            Arc::new(MemoryIO::new()),
            "row-security-materialized-view.db",
            crate::OpenFlags::default(),
            crate::DatabaseOpts::new().with_views(true),
            None,
            Arc::new(EngineDialect),
        )
        .unwrap();
        let conn = db.connect(crate::ConnectOpts::new()).unwrap();
        conn.execute("CREATE TABLE t (x)").unwrap();
        conn.execute("CREATE MATERIALIZED VIEW mv AS SELECT x FROM t")
            .unwrap();
        let mv = || ast::QualifiedName::single(ast::Name::exact("mv".to_string()));
        let statements = [
            ast::Stmt::AlterTable(ast::AlterTable {
                name: mv(),
                body: ast::AlterTableBody::EnableRowSecurity,
            }),
            ast::Stmt::CreatePolicy(Box::new(ast::CreatePolicy {
                policy_name: ast::Name::exact("p".to_string()),
                tbl_name: mv(),
                permissive: true,
                command: ast::PolicyCommand::All,
                roles: vec![ast::Grantee::Public],
                using_expr: None,
                check_expr: None,
            })),
        ];

        for statement in statements {
            let sql = statement.to_string();
            let error = conn
                .prepare_translated_cmd(ast::Cmd::Stmt(statement), &sql)
                .unwrap_err();

            assert!(
                error.to_string().contains("\"mv\" is not a table"),
                "{error}"
            );
        }
    }

    #[test]
    fn enabling_row_security_fails_under_a_dialect_without_it() {
        let db = Database::open_file(
            Arc::new(MemoryIO::new()),
            ":memory:",
            Arc::new(EngineDialect),
        )
        .unwrap();
        let conn = db.connect(crate::ConnectOpts::new()).unwrap();
        conn.execute("CREATE TABLE t (x)").unwrap();
        let enable = ast::Cmd::Stmt(ast::Stmt::AlterTable(ast::AlterTable {
            name: ast::QualifiedName::single(ast::Name::exact("t".to_string())),
            body: ast::AlterTableBody::EnableRowSecurity,
        }));

        let error = conn
            .prepare_translated_cmd(enable, "ALTER TABLE t ENABLE ROW LEVEL SECURITY")
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("row-level security is not supported"),
            "{error}"
        );
    }
}
